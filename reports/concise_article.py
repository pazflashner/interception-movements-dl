"""Eight-page scientific paper plus a separately paginated evidence appendix."""
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.lib.units import mm
from reports.article_layout import Article, benchmark_rows, probe_rows, number, pvalue, target_label

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "studies/review_corrected_evaluation/results"
CONTROLS = RESULTS / "review_controls"

REFERENCES = [
    "[1] Brenner E, Smeets JBJ. Continuously updating one's predictions underlies successful interception. <i>J Neurophysiol.</i> 2018;120:3257-3274. doi:10.1152/jn.00517.2018.",
    "[2] Slowinski P et al. Dynamic similarity promotes interpersonal coordination in joint action. <i>J R Soc Interface.</i> 2016;13:20151093. doi:10.1098/rsif.2015.1093.",
    "[3] Kingma DP, Welling M. Auto-Encoding Variational Bayes. ICLR, 2014. arXiv:1312.6114.",
    "[4] Sohn K, Lee H, Yan X. Learning Structured Output Representation using Deep Conditional Generative Models. <i>NeurIPS.</i> 2015;28.",
    "[5] Flash T, Hogan N. The coordination of arm movements: an experimentally confirmed mathematical model. <i>J Neurosci.</i> 1985;5:1688-1703. doi:10.1523/JNEUROSCI.05-07-01688.1985.",
    "[6] Friedman J. <i>submovements.</i> github.com/JasonFriedman/submovements. Revision 9c2f40ccc922d542242329c46cfd524c21188b4a; locally adapted, GPL-3.0.",
    "[7] Rohrer B, Hogan N. Avoiding spurious submovement decompositions II: a scattershot algorithm. <i>Biol Cybern.</i> 2006;94:409-414. doi:10.1007/s00422-006-0055-y.",
    "[8] Gretton A et al. A Kernel Two-Sample Test. <i>J Mach Learn Res.</i> 2012;13:723-773. jmlr.org/papers/v13/gretton12a.html.",
    "[9] Szekely GJ, Rizzo ML. Energy statistics: a class of statistics based on distances. <i>J Stat Plan Inference.</i> 2013;143:1249-1272. doi:10.1016/j.jspi.2013.03.018.",
    "[10] Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. <i>J R Stat Soc B.</i> 1995;57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x.",
    "[11] Bengio Y, Grandvalet Y. No unbiased estimator of the variance of K-fold cross-validation. <i>J Mach Learn Res.</i> 2004;5:1089-1105. jmlr.org/papers/v5/grandvalet04a.html.",
    "[12] SciPy developers. <i>scipy.stats</i> documentation: ks_2samp and wilcoxon; numerical-tie guidance. docs.scipy.org/doc/scipy/reference/stats.html. Main paired tests executed with SciPy 1.16.3.",
]


def controls():
    return {name: pd.read_csv(CONTROLS / (name + ".csv")) for name in [
        "fingerprint_summary", "fingerprint_paired", "direct_context_summary",
        "matched_component_summary", "matched_component_sampling", "matched_component_diagnostics",
        "matched_component_sensitivity_summary", "matched_component_paired"]}


def fingerprint_rows(c):
    rows = [["n", "Fingerprint supplied", "Mean KS", "Energy", "MMD<super>2</super>"]]
    names = {"own": "Correct participant", "population": "Training-participant average", "wrong": "Other participants (mean distance)"}
    f = c["fingerprint_summary"].set_index(["latent_dim", "arm"])
    for n in [3, 8]:
        for arm in ["own", "population", "wrong"]:
            r = f.loc[(n, arm)]
            rows.append([n, names[arm], number(r.mean_ks), number(r.energy_distance), number(r.mmd_rbf)])
    return rows


def timing_rows(tables):
    rows = [["Head comparison", "n", "Endpoint", "CVAE ms", "Spline ms", "Adjusted p"]]
    for _, r in tables["timing_fairness_paired"].iterrows():
        rows.append(["Common MLP" if r.comparison == "common_mlp" else "Validation calibration",
                     int(r.latent_dim), r.endpoint, number(r.cvae_mean_ms, 1), number(r.spline_mean_ms, 1), pvalue(r.p_fdr_bh)])
    return rows


def component_rows(c):
    f = c["matched_component_sampling"]
    rows = [["Component discrepancy", "n=3", "n=8", "Sampling reference"]]
    names = {"ks_mj_fit_error": "Fit error KS", "ks_mj_first_duration_s": "First duration KS",
        "ks_mj_first_amplitude": "First displacement amplitude KS", "ks_mj_secondary_amplitude_fraction": "Secondary amplitude fraction KS",
        "ks_mj_mean_overlap_pct": "Overlap KS", "count_total_variation": "Count total variation", "count_jsd": "Count JSD"}
    for key, name in names.items():
        s = f[f.metric == key].set_index("latent_dim")
        rows.append([name, number(s.loc[3, "observed"]), number(s.loc[8, "observed"]), number(s.loc[3, "reference_mean"])])
    return rows


def scientific_report(output, tables, fingerprints, figures, example_meta):
    c = controls()
    a = Article(output, "Low-Dimensional Generative Models of Human Interception Movements",
        "Simaan Libbiss and Paz Flashner<br/>Workshop on Deep Learning, Tel Aviv University<br/>"
        "Research supervision: Prof. Jason Friedman | Course advisor: Moni Shahar")
    a.h("Abstract")
    a.p("We study whether a compact representation of a participant's interception trials can generate the distribution of that participant's remaining movements. Five model families are evaluated on 4,732 trials from 28 participants using four participant-held-out folds. Spline+PCA reconstructs trajectories more accurately than a conditional variational autoencoder (CVAE), whereas CVAE has lower original-head timing errors and several lower distribution distances. A same-decoder control shows that the correct participant fingerprint improves generation over population and other-participant fingerprints at both three and eight latent dimensions. However, direct context measurements outperform latent probes on all 14 tested behavioural summaries under mean absolute error. Common timing heads reduce the CVAE initiation-time advantage, and minimum-jerk decomposition remains a secondary, assumption-sensitive evaluation. These results support useful participant information in the learned generator, while falling short of a validated low-dimensional account of individual movement strategy.")
    a.h("1 Introduction")
    a.p("Interception requires reaching an interception region at an appropriate time while the target moves [1]. The modelling objective is to reproduce distributions of initiation time, movement time and detailed kinematics across people. Movement can begin while information is still being gathered; describing a trajectory does not by itself identify the underlying decision process. Individual motor signatures motivate compact representations [2], although that work concerns a different task.")
    a.p("We distinguish reconstruction of an observed trial, prediction of its physical timing from its observed shape, and generation of new trials after enrolment from a held-out participant's context recordings. Only the third directly addresses distribution generation. Two or three latent coordinates are the desired compact interface; eight coordinates provide a higher-capacity comparison.")
    a.h("2 Data and study cohort")
    a.p("Participants moved a finger toward a fixed square as a circle approached it horizontally, aiming to arrive while the circle was inside. Six target start locations combine three distance/speed categories with left/right sides; speed varies continuously within category. Finger position was recorded in 3D at 240 Hz using a Polhemus Liberty tracker; target trajectories were sampled at 60 Hz. Following Prof. Friedman's instructions, only condition 2 (free eye movements) was analysed.")
    a.p("Of 4,763 available condition-2 recordings, four missing-arrival timeouts and 27 trials arriving more than 1 s after the target window were excluded, leaving 4,732 trials from 28 participants. The original project slides mention 29 participants; this report describes the 28 available in the analysed files. The retained outcomes are 3,006 Success, 326 Too late, 48 Too early and 1,352 fixation warnings. Outcome labels are not encoder inputs; early outcomes and warnings remain included provisionally.")

    a.page()
    a.h("3 Methods")
    a.h("3.1 Temporal and spatial representation", True)
    a.p("Frame counters establish time; duplicate positions are averaged and missing counters interpolated. Marker 5 identifies target appearance. MAT target trajectories identify motion onset, 0.20-0.50 s later (median 0.35 s), which anchors the model window. Positions are fourth-order, zero-phase Butterworth filtered at 10 Hz, translated to the initial window position and resampled to 100 phase points. The final model uses lateral x and forward y; the original 3D recordings remain available.")
    a.p("The model window runs from target motion to the final recorded finger sample, retaining post-motion waiting. The endpoint is an arrival proxy, on average 26.2 ms before MAT pressedTime (SD 3.6 ms). Finger onset is the first post-motion 3D speed crossing above 5 tracker units/s sustained for three frames. Initiation and movement durations are derived from these indices. Physical durations are withheld from the encoder and predicted separately; relative waiting remains visible in the phase-normalized shape. Units and event alignment require domain confirmation.")
    a.h("3.2 Models and training", True)
    a.p("The CVAE [3,4] encodes a 100 x 2 trajectory and five condition values (three start-category indicators, side and exact target speed) through two 256-unit ReLU layers into a diagonal Gaussian posterior. The decoder maps a latent draw and conditions to shape and two log-duration outputs. Its loss sums standardized coordinate squared errors, 20 times standardized log-timing squared errors and a KL term whose weight increases from 0 to 1 over 50 epochs. This is a weighted reconstruction objective, not a likelihood for the reported behavioural feature distributions.")
    a.table([
        ["Family", "Dimensions", "Role in the comparison"],
        ["CVAE", "2, 3, 4, 8", "Conditional variational shape/timing model"],
        ["Spline+PCA", "2, 3, 4, 8", "Cubic splines, five interior knots; train-fitted PCA; Ridge timing head"],
        ["Conditional AE", "3, 8", "Conditional network without stochastic bottleneck or KL"],
        ["Unconditional VAE", "3, 8", "Variational network with task-condition input suppressed"],
        ["Condition Ridge", "No fingerprint", "Condition-only shape/timing regression plus training residual sampling"],
    ], "<b>Table 1.</b> Model families. Matched latent size does not equalize parameter counts, objectives or timing-head capacity.", [35*mm,28*mm,103*mm])
    a.p("Training uses participant-balanced sampling, Adam (initial learning rate 0.001; batch 64), gradient clipping at 5, validation learning-rate reduction and early stopping (patience 25), with a 150-epoch limit. Five of the 96 neural runs reached this limit. Scaling statistics are fitted on training data. The classical and neural settings are detailed in Appendix S1.")
    a.h("3.3 Participant-held-out evaluation", True)
    a.p("Each of four fixed folds uses 17 training, four validation and seven test participants; every participant is tested once. Seeds 42, 43 and 44 vary neural training randomness. The matrix has 48 CVAE, 24 conditional-AE, 24 unconditional-VAE, 16 spline and 12 condition-Ridge evaluations. These are all declared folds and seeds, not every possible partition. Test participants contribute disjoint, approximately half-session context/query sets stratified by start category and side. Their query trajectories never determine the fingerprint.")

    a.page()
    a.h("4 Reconstruction, timing and distribution fidelity")
    a.p("Reconstruction uses posterior means; trajectory MSE averages coordinate errors within trials and participants. Timing MAE averages absolute errors within participants, then participants equally. Generation draws 120 trials per participant around the context-mean latent code using a shared training-derived covariance. Query condition metadata determine the sampled task mixture; query shapes and timing labels are withheld. This evaluates enrolled participants under a known task mixture, not zero-shot prediction.")
    a.p("Mean KS averages two-sample KS statistics over 11 features and participants: initiation/movement time, peak speed, relative peak time, path length, straight-line distance, curvature, lateral deviation, heuristic speed-peak count and x/y endpoints. It is a discrepancy, not accuracy or a combined p-value. MMD<super>2</super> [8] and energy [9] compare standardized multivariate features using one training-only reference per fold; Appendix S1 gives the estimators.")
    a.table(benchmark_rows(tables["oof"], fingerprints), "<b>Table 2.</b> Out-of-fold, participant-balanced means. Neural results average three seeds. Lower error/distance is better. MSE is in tracker units squared; timing in ms. Enrolment is closed-set classification among seven enrolled people (chance 14.3%).", [33*mm,8*mm,17*mm,23*mm,23*mm,18*mm,21*mm,23*mm])
    a.p("Spline has lower reconstruction MSE than CVAE at n=3 (0.097 versus 0.177) and n=8 (0.023 versus 0.032). CVAE has lower original-head timing MAE, mean KS and MMD<super>2</super> at these dimensions. The paired CVAE-spline contrasts shown in Appendix S2 survive BH correction across the 56 main tests. They establish an endpoint-dependent tradeoff, not a uniformly superior model.")
    a.figure(figures["benchmark"], "<b>Figure 1.</b> Selected benchmark endpoints; all panels show errors or discrepancies. The complete dimension sweep is in Appendix S2.", 55*mm)
    a.p("At n=3, CVAE does not improve mean KS, MMD or energy over condition Ridge after correction; at n=8 it improves all three. Conditional AE at n=8 reconstructs better than CVAE (MSE 0.026 versus 0.032), while CVAE improves KS (0.219 versus 0.259) and MMD<super>2</super> (0.067 versus 0.111). Unconditional VAE remains competitive, so neither conditioning nor the variational objective is uniformly beneficial.")

    a.page()
    a.h("5 Does the correct person's fingerprint help?")
    a.p("A post-review control isolates the fingerprint supplied to the same trained CVAE. For each of the 24 n=3/8 fold/seed checkpoints, we keep the decoder, 120 condition draws, latent-noise draws, shared covariance and distance reference fixed. We substitute (i) the participant's own context centroid, (ii) the equal-person average of the 17 training-context centroids, or (iii) each of the other six held-out context centroids. The wrong-person score averages six separate distances; it does not pool their generated distributions.")
    a.table(fingerprint_rows(c), "<b>Table 3.</b> Same-decoder fingerprint controls. Values average seeds within people, then all 28 people. Own-fingerprint generation reproduces the existing benchmark before controls are compared.", [10*mm,75*mm,27*mm,27*mm,27*mm])
    largest_p = c["fingerprint_paired"].p_fdr_bh.max()
    a.p(f"The correct fingerprint improves all three distances against both controls at both dimensions (12 comparisons; all BH-adjusted p &lt; {largest_p + 0.000001:.4f}). At n=3, mean KS improves over the population centre for 22/28 participants and over other-person centres for 27/28; at n=8 the corresponding counts are 27/28 and 27/28. Mean KS falls from 0.311 to 0.278 against the population control at n=3, and from 0.290 to 0.219 at n=8.")
    a.figure(figures["fingerprint_control"], "<b>Figure 2.</b> Participant-level change in mean KS when using the correct rather than a control fingerprint. Each point is one participant after averaging seeds; positive values favour the correct fingerprint. Lines mark zero and the participant mean.", 85*mm)
    a.p("This supports a specific claim: personal context contains information that improves this generator's within-session distribution predictions. It does not establish that the fingerprint recovers a cognitive strategy, stays stable across sessions, or reproduces every behavioural distribution. A useful fingerprint control can coexist with incomplete absolute fidelity and with competitive unconditional models.")
    a.h("5.1 Statistical scope", True)
    a.p("Paired two-sided Wilcoxon tests use one seed-averaged value per participant. The 56 main and 12 fingerprint tests use Pratt zero handling, SciPy 1.16.3 method='auto', and differences rounded to 12 decimal places before ranking [12]; effect-size means remain unrounded. BH correction [10] is applied separately to declared comparison families. Overlapping training folds mean participant errors are not fully independent [11]; these post-review controls support a within-cohort comparison, not an independent population replication.")

    a.page()
    a.h("6 Timing sensitivity and behavioural prediction")
    a.h("6.1 Comparing timing heads", True)
    a.p("The original CVAE timing head is nonlinear; spline uses linear Ridge. A sensitivity fits the same two-layer 64-unit MLP to frozen CVAE or spline codes plus conditions, using training/validation data only. A second sensitivity fits a positive multiplicative timing calibration to validation MAE. These comprise 48 additional small MLP fits and calibration checks, not retraining of the generative models.")
    a.table(timing_rows(tables), "<b>Table 4.</b> Timing sensitivities; BH correction covers eight paired comparisons. Calibration changes these sensitivity predictions only.", [46*mm,8*mm,29*mm,26*mm,26*mm,31*mm])
    a.p("With a common MLP, the CVAE initiation-time advantage is smaller but remains detectable; movement-time differences do not survive correction. CVAE codes were learned with timing supervision and spline codes were not, so the common head does not isolate representation geometry from the training objective. Validation calibration likewise cannot assign a percentage of the advantage to a particular cause.")
    a.h("6.2 Compact probes versus direct context measurements", True)
    a.p("Validation-tuned Ridge probes predict means and SDs of seven query behaviours from context fingerprints, refitting on 21 development participants before testing. For CVAE n=3, initiation-mean pooled R<super>2</super> is 0.723 and peak-speed-mean R<super>2</super> is 0.356; 10/14 pooled and 12/14 mean-fold scores are negative. Pooled and mean-fold R<super>2</super> have different denominators and are reported separately in Appendix S4.")
    direct = c["direct_context_summary"].set_index(["latent_dim", "target", "arm"])
    rows = [["Query-summary target", "CVAE n=3 MAE", "CVAE n=8 MAE", "Direct context MAE", "Development constant MAE"]]
    for target, label, multiplier in [("initiation_time_s_mean", "Mean initiation (ms)", 1000), ("peak_speed_tracker_units_s_mean", "Mean peak speed (tracker units/s)", 1)]:
        rows.append([label, *[number(direct.loc[(n,target,"cvae_probe"),"mae"]*multiplier,2) for n in [3,8]], number(direct.loc[(3,target,"direct_context"),"mae"]*multiplier,2), number(direct.loc[(3,target,"development_constant"),"mae"]*multiplier,2)])
    a.table(rows, "<b>Table 5.</b> Summary prediction controls on the same query targets. The constant uses the 21 development participants; direct context copies the corresponding measured context mean or SD. Values average seed-specific MAEs.", [47*mm,28*mm,28*mm,31*mm,32*mm])
    a.p("Direct context has lower average MAE on all 14 targets at both dimensions. Its pooled R<super>2</super> is 0.912 for mean initiation and 0.976 for mean peak speed. This is a descriptive comparison, not a new significance claim. Direct measurement retains the target-specific summaries (14 values jointly), whereas the latent probe shares three or eight coordinates across targets. Thus personal information helps generation, but the tested compression does not improve these summary predictions over measuring the available context directly.")

    a.page()
    a.h("7 Minimum-jerk component fidelity")
    a.p("As a secondary description, we approximate movement velocity using one to four overlapping minimum-jerk pulses [5-7]. Each pulse has an onset, duration and planar displacement. Counts are selected by normalized fit error (smallest k at error at most 0.05, then below 0.10, otherwise minimum error); BIC is an alternative. The implementation adapts Friedman's code and is not identical to the published scattershot procedure. Components and heuristic speed peaks are different quantities; neither directly counts decisions.")
    a.p("The original recorded/generated fits had unequal optimization budgets and input processing. The post-review rerun applies the same downstream operator to the recorded model-target window and generated window: 100-point go-to-end input, timing-based extraction of 100 movement points, restoration of physical sample rate, low-pass filtering and two optimizer restarts with at most 400 function evaluations. All 2,376 query recordings and the same 1,680 generated samples (10 per person/seed/dimension) are fitted. Upstream acquisition filtering is retained in the recorded target; this is a comparison of the model representation, not unprocessed biological components.")
    a.table(component_rows(c), "<b>Table 6.</b> Matched-procedure component discrepancies. The empirical sampling reference uses 500 replicates at each participant's query size and ten generated samples, averaging three seed-like draws. It is a plug-in reference, not a formal goodness-of-fit p-value or hard floor.", [67*mm,30*mm,30*mm,39*mm])
    diag = c["matched_component_diagnostics"].set_index("kind")
    a.p("Count total variation falls from historical 0.353/0.355 to 0.214/0.232 at n=3/8, weakening the earlier mismatch magnitude. All seven discrepancies remain above the empirical reference's 95th percentile, which excludes fitting and training uncertainty. All 4,056 selected fits converged; 505 fits had a nonconverged alternative order (Appendix S5). These are evaluation changes, not new neural training.")
    a.figure(figures["matched_components"], "<b>Figure 3.</b> Legacy versus matched-procedure count discrepancies and selected-count proportions. The matched results are separate from the main 11-feature benchmark.", 53*mm)
    sens = c["matched_component_sensitivity_summary"].set_index("kind")
    significant = int((c["matched_component_paired"].p_fdr_bh < .05).sum())
    a.p(f"A fixed sensitivity subset (56 recorded and 112 generated trajectories) uses four restarts and 700 evaluations; selected-count agreement with the matched base is {100*sens.loc['recorded','count_agreement']:.1f}% and {100*sens.loc['generated','count_agreement']:.1f}%, respectively. Of nine paired n=3 versus n=8 component endpoints, {significant} survive BH correction. Historical 167 ms-bound and 5 Hz sensitivities also changed counts materially (Appendix S5). The component results remain conditional on fitting assumptions and finite sampling.")

    a.page()
    a.h("8 Discussion and limitations")
    a.p("The study supports an endpoint-dependent conclusion. Spline remains the stronger shape compressor; CVAE improves several distribution endpoints over spline and, at n=8, condition Ridge. Correct-person fingerprint controls establish useful participant information in this generator. Direct context summaries nevertheless outperform the tested latent probes, so these results do not establish a sufficient two- or three-coordinate account of a person's behavioural distributions.")
    a.h("8.1 Conditioning, clustering and interpretation", True)
    a.p("The in-sample trajectory K-means analysis gives ARI=0.052 and permutation p=0.00498 (200 permutations), indicating weak non-random structure rather than clean participant clusters. Closed-set enrolment reaches 40.9% at n=3 and 50.7% at n=8 against seven enrolled candidates; this is distinct from generation and cannot establish cross-session stability. The condition-stratum trajectory diagnostic gives Holm-adjusted p=1.000 and 0.132 at n=3/8, and none of 44 feature-level comparisons survives BH correction. Therefore the condition sliders remain exploratory. Latent-feature heatmaps show within-model associations, not independently validated strategy axes (Appendix S6).")
    a.h("8.2 Event definitions and retained early movement", True)
    a.p("There are 357 zero-initiation labels (7.5%): the post-target-motion speed threshold is immediately satisfied, with no missing-onset fallback among these trials. Of these, 109 show at least 50 ms of pre-motion threshold exceedance in filtered planar speed, and 100 in raw speed. Movement before target motion is distinct from starting while the moving target can supply information. These observations are retained and do not determine their cause; the encoder still begins at target motion. Event synchronization, endpoint validity and intended movement permissions need Prof. Friedman's confirmation.")
    a.figure(figures["event_excerpt"], "<b>Figure 4.</b> An audited example with raw and filtered planar speed around target-motion time. Threshold exceedance is descriptive evidence, not a diagnosis of noise or anticipation. Full examples and selection rules are in Appendix S7.", 58*mm)
    a.p("Rare timing failures also remain visible: CVAE initiation predictions reach about 11.5 s, without post-hoc clipping to improve the benchmark. The 74.6% statistic associated with zero-onset trials is a share of centred log-target variance, not observed trained-model loss or a demonstrated cause of outliers. The condition-2 fixation warning, the 1 s late exclusion, tracker units, onset threshold and movement-window choice remain acquisition/analysis assumptions, not established behavioural facts.")
    a.p("All participants come from one dataset; folds share training people. No learning-curve experiment establishes that participant count caused model failure, and no independent cohort validates the learned representation. Frozen-model controls and matched refits are explicitly post-review analyses. Stronger claims about cognition, stable identity or extrapolation require further targeted evidence rather than more favourable examples.")

    a.page()
    a.h("9 Conclusion")
    a.p("A participant's context-derived fingerprint improves generation with the tested CVAE, while reconstruction, summary prediction and component fidelity expose complementary limitations. The results justify a compact generative interface as an experimental model and support further evaluation of movement distributions. They do not yet justify a complete or uniquely interpretable model of individual interception strategy.")
    a.h("Reproducibility and acknowledgments")
    a.p("The evaluation covers all 124 declared model runs and preserves the original checkpoints, predictions and component fits. The matched component refits and fingerprint controls reuse these checkpoints; they do not retrain the neural models or change exclusions. Fold assignments, seeds, settings, checkpoint hashes and participant-level result tables accompany the code and appendix. Source histories record dirty-worktree states for two model families, so reproducibility is tied to saved artifacts and hashes rather than an unsupported clean-commit claim.")
    a.p("We thank Moni Shahar for project guidance and Prof. Jason Friedman for the experiment, data and methodological resources. AI tools assisted software development and report preparation; the authors remain responsible for understanding, verification and interpretation. A separate appendix contains the full dimension sweep, paired tests, all behavioural probe targets, convergence diagnostics, sensitivity checks, representative trajectories and execution instructions.")
    a.h("References")
    for ref in REFERENCES: a.p(ref, "caption")
    a.finish()


def appendix_report(output, tables, fingerprints, figures, example_meta):
    c = controls()
    a = Article(output, "Supplementary Appendix", "Low-Dimensional Generative Models of Human Interception Movements<br/>Simaan Libbiss and Paz Flashner")
    a.p("This appendix accompanies the eight-page scientific report. It supplies extended methods, complete comparisons and diagnostic results. Historical component outputs and matched-procedure refits are explicitly separated; neither is evidence of a validated cognitive strategy.")
    a.h("S1 Extended methods and reproducibility")
    a.p("CVAE position input is 200 values; five condition values enter encoder and decoder. Physical timing is transformed as log(t+0.001), standardized with training statistics, and inverted as max(exp(u)-0.001,0). The loss sums 200 standardized coordinate errors and 20 times two timing errors, plus the KL penalty. Adam begins at 0.001; ReduceLROnPlateau halves learning rate after 15 validation plateaus. Early stopping uses patience 25 and at most 150 epochs; five of 96 neural histories hit the cap. Validation selection uses the configured variational validation objective, whereas reported reconstruction uses posterior means.")
    a.p("Spline fitting uses cubic B-splines with five interior knots and nine coefficients per coordinate (18 in total), followed by training-only PCA to the tested dimensions. Physical timings are excluded from spline-PCA inputs. Validation selects Ridge regularization for the timing head. The spline generator samples around context-mean coefficients with training-derived variation. Conditional AE removes stochastic latent sampling and KL; unconditional VAE suppresses condition inputs. Condition Ridge uses conditions alone and samples training residuals by task stratum.")
    a.p("A CVAE fingerprint is the context mean of posterior means. The shared latent covariance combines within-participant variation of training posterior means and average diagonal posterior uncertainty, plus a small diagonal regularizer. A shared latent covariance need not produce equal behavioural variances after nonlinear decoding. Mean-plus-SD fingerprints are separate higher-dimensional probe sensitivities.")
    a.p("The four outer folds test every person once (17/4/7 train/validation/test); neural seeds are 42, 43 and 44. Context/query splitting uses seed 2026 plus a deterministic subject offset, with approximately half the trials in each start-category/side stratum. Generated task draws use the target person's query condition metadata, not query shapes or timing labels. An enrolment score is classification among seven known context centroids after context-derived scaling; nominal chance is 1/7.")
    a.page(); a.h("S1.1 Distance estimators and inference", True)
    a.figure(figures["pipeline"], "<b>Figure S1.</b> Data, model and evaluation flow; the post-review controls extend the evaluation of frozen models.", 85*mm)
    a.p("Continuous multivariate features use training standard deviations, count scales are at least one count and constant scales are one. A training median-distance RBF bandwidth is fixed per fold using at most 512 examples. The unbiased squared-MMD U-statistic excludes within-sample diagonals and may be negative in finite samples. Energy is 2 mean||X-Y|| - mean||X-X'|| - mean||Y-Y'|| including diagonals and with no square root. Mean KS weights 11 feature statistics equally; it is not a combined hypothesis test. Heuristic speed peaks use 10% prominence and 50 ms separation; minimum-jerk counts use fitted components.")
    a.p("Main comparisons average seeds per participant, then apply two-sided Pratt Wilcoxon tests with 12-decimal difference rounding, SciPy 1.16.3 method='auto'. All 56 main significance decisions are unchanged by this numerical correction. BH families are separate: 56 main contrasts, eight timing sensitivities, 12 fingerprint controls and nine matched-component dimension comparisons. Other historical sensitivity families retain their recorded conventions. Cross-validation dependence limits inference. Core execution used Python 3.13, NumPy 2.3.4, pandas 2.3.3 and SciPy 1.16.3.")
    a.page(); a.h("S2 Complete model comparison")
    a.table(benchmark_rows(tables["oof"], fingerprints, True), "<b>Table S1.</b> Complete dimension sweep; definitions and units follow main Table 2.", [33*mm,8*mm,17*mm,23*mm,23*mm,18*mm,21*mm,23*mm])
    rows = [["n", "Endpoint", "CVAE", "Spline", "CVAE lower /28", "Adjusted p"]]
    metrics = {"trajectory_mse": "Trajectory MSE", "initiation_time_mae_ms": "Initiation MAE", "movement_time_mae_ms": "Movement MAE", "mean_ks": "Mean KS", "mmd_rbf": "MMD squared"}
    for _, r in tables["paired"].query("comparator == 'spline_pca'").iterrows():
        if r.metric in metrics: rows.append([int(r.latent_dim), metrics[r.metric], number(r.cvae_mean), number(r.comparator_mean), int(r.cvae_better_participants), pvalue(r.wilcoxon_p_fdr_bh)])
    a.table(rows, "<b>Table S2.</b> CVAE-spline contrasts within the 56-comparison BH family; timing values are ms. The complete 56-row CSV accompanies this appendix.", [10*mm,43*mm,26*mm,26*mm,31*mm,30*mm])
    a.page(); a.h("S3 Complete fingerprint controls")
    rows = [["n", "Control", "Endpoint", "Control - own", "Own lower /28", "Adjusted p"]]
    for _, r in c["fingerprint_paired"].iterrows():
        rows.append([int(r.latent_dim), r.control, {"mean_ks":"Mean KS","energy_distance":"Energy","mmd_rbf":"MMD squared"}[r.metric], number(r.control_minus_own), int(r.own_better_n), pvalue(r.p_fdr_bh)])
    a.table(rows, "<b>Table S3.</b> Full same-decoder control family (12 tests). Wrong-person distances average the six other test-context donors before seed averaging.", [10*mm,28*mm,37*mm,31*mm,30*mm,30*mm])
    a.p("Every own-fingerprint result was checked against the existing saved distribution result before a control contrast was computed. Raw output retains each donor identity, context/query sample counts, per-feature distances and run identifiers. This control changes only the supplied latent centre; it does not retrain a model or use query trajectories to enrol a participant.")
    a.figure(figures["capacity"], "<b>Figure S2.</b> Capacity and enrolment across latent dimensions; error bars describe fold/seed spread, not independent-cohort confidence intervals.", 85*mm)
    a.page(); a.h("S4 All behavioural prediction targets")
    a.table(probe_rows(tables["probes"]), "<b>Table S4.</b> All 14 CVAE mean-fingerprint query-summary probes; each R-squared statistic is averaged over seeds. Mean-fold and pooled statistics are not interchangeable.", [58*mm,27*mm,27*mm,27*mm,27*mm])
    a.p("Ridge regularization is chosen on validation participants, then the probe refits on the 21 development participants. The historical constant is based on the 17 training participants. The added constant instead uses the 21 development query-summary targets, matching access at probe refitting. Direct context copies each measured context mean or sample SD to predict the same quantity on query trials. It uses no query outcomes; taken jointly it retains 14 target-specific quantities rather than a shared n-dimensional code.")
    a.page(); a.h("S4.1 Direct-context MAE on every target")
    d = c["direct_context_summary"].set_index(["latent_dim", "target", "arm"])
    rows = [["Target", "CVAE n=3", "CVAE n=8", "Direct context", "21-person constant"]]
    for target in c["direct_context_summary"].target.unique():
        mult = 1000 if target.startswith(("initiation_time_s", "movement_time_s")) else 1
        label = target_label(target) + (" [ms]" if mult == 1000 else "")
        rows.append([label, number(d.loc[(3,target,"cvae_probe"),"mae"]*mult,3), number(d.loc[(8,target,"cvae_probe"),"mae"]*mult,3), number(d.loc[(3,target,"direct_context"),"mae"]*mult,3), number(d.loc[(3,target,"development_constant"),"mae"]*mult,3)])
    a.table(rows, "<b>Table S5.</b> Mean absolute errors on all summary targets. Timing is ms; peak speed is tracker units/s; path length and lateral deviation use tracker units; curvature and counts are dimensionless. Descriptive comparison, without an additional hypothesis-test family.", [58*mm,27*mm,27*mm,27*mm,27*mm])
    a.p("The direct-context baseline has lower MAE than the CVAE probe on all 14 targets at each dimension. This does not establish equivalence of information budgets or superiority for full trajectory generation. All model-family probe results and mean-plus-SD variants remain available in the machine-readable tables.")
    a.h("S4.2 Timing calibration details", True)
    a.p("The common timing MLP has two 64-unit hidden layers, participant weighting, training-only input/log-target scaling, validation early stopping (patience 40) and at most 400 epochs. There are 48 such fits. The multiplicative calibration is positive and selected using validation MAE for each timing endpoint; it does not alter the main trajectory generator. Main Table 4 includes all eight paired comparisons.")
    a.page(); a.h("S5 Component refitting and sensitivity")
    a.table(component_rows(c), "<b>Table S6.</b> Matched-procedure distances and empirical sampling reference; repeated here beside diagnostics.", [67*mm,30*mm,30*mm,39*mm])
    rows = [["Input", "Fits", "Selected converged", "All candidates converged", "Legacy count agreement"]]
    for _, r in c["matched_component_diagnostics"].iterrows():
        rows.append([r.kind, int(r.n), int(r.selected_converged_n), int(r.all_candidates_converged_n), number(100*r.count_agreement_with_legacy,1)+"%"])
    a.table(rows, "<b>Table S7.</b> Optimizer diagnostics. A completed procedure can include nonconverged candidate orders; convergence is not a guarantee of a global optimum.", [30*mm,23*mm,37*mm,41*mm,35*mm])
    a.p("The component basis is (30u squared - 60u cubed + 30u to the fourth)/duration, multiplied by x/y displacement and zero outside the component interval. The objective includes x/y residuals and the difference between norms of summed velocities. It pads the observation with zeros beyond the recorded interval. Bounds allow durations from 0.100 to 1.000 s; the 0.050 s setting is an indexed absolute onset lower-bound step, not a pairwise onset gap. The BIC calculation includes correlated speed and zero padding, so it is a model-order heuristic here.")
    a.p("Recorded inputs in this rerun are the canonical filtered 100-point go-window targets; generated inputs are decoder outputs on the same grid. Both use the same timing-based crop, low-pass filter and two-restart/400-evaluation fitting code. Physical movement timing is measured for recordings and predicted for generated samples. The pre-acquisition/target filtering history is not erased. This control therefore evaluates the represented trajectories with a common downstream procedure; it does not validate the inferred components as distinct biological commands.")
    a.page(); a.h("S5.1 Dimension comparisons and optimizer sensitivity")
    rows = [["Endpoint", "n=3", "n=8", "n=8 lower /28", "Adjusted p"]]
    labels = {"count_total_variation":"Count TV","count_jsd":"Count JSD","count_total_variation_bic":"BIC count TV","count_jsd_bic":"BIC count JSD"}
    for _, r in c["matched_component_paired"].iterrows():
        label = labels.get(r.metric, r.metric.replace("ks_mj_", "KS ").replace("_", " "))
        rows.append([label, number(r.n3_mean), number(r.n8_mean), int(r.n8_better_n), pvalue(r.p_fdr_bh)])
    a.table(rows, "<b>Table S8.</b> Nine paired matched-component dimension contrasts, with one BH family.", [58*mm,27*mm,27*mm,27*mm,27*mm])
    rows = [["Input", "Subset n", "Count agreement", "Base selected converged", "High-budget converged"]]
    for _, r in c["matched_component_sensitivity_summary"].iterrows():
        rows.append([r.kind, int(r.n), number(100*r.count_agreement,1)+"%", int(r.base_selected_converged_n), int(r.high_budget_selected_converged_n)])
    a.table(rows, "<b>Table S9.</b> First two query trials and first two generated samples per person, n=3/8 seed42; four restarts and 700 evaluations compared with the matched base.", [30*mm,23*mm,36*mm,39*mm,38*mm])
    a.p("Historical count TV was 0.353/0.355 at n=3/8. Those recorded fits used two restarts and 400 evaluations on already-filtered 240-Hz movement slices; generated fits used one restart and 300 evaluations. On 56 historical recorded trials, four-restart/700-evaluation counts agreed in 100%, while 167 ms duration/onset-bound and 5 Hz changes agreed in 55.4% and 51.8%. These historical sensitivities do not certify the new procedure; they show why fitting assumptions must remain visible.")
    a.p("The matched sampling reference resamples each observed query component distribution 500 times: an independent query-sized and ten-sample draw, with three seed-like replicates averaged before equal-person aggregation. It conditions on the fitted empirical distribution and does not include fitting or training uncertainty. No reference value is a hard threshold for a satisfactory model.")
    a.page(); a.h("S6 Conditioning and latent associations")
    a.figure(figures["condition"], "<b>Figure S3.</b> Condition-stratum trajectory diagnostic. Holm-adjusted p=1.000 (n=3) and 0.132 (n=8); no feature-level contrast among 44 survives BH correction.", 92*mm)
    a.figure(figures["associations"], "<b>Figure S4.</b> Within-decoder Spearman correlations for 500 training-centred latent draws at fixed sp=2, side=1 and speed 0.635 screen widths/s; CVAE n=3, fold0, seed42. Correlated draws and coordinate non-identifiability limit interpretation.", 80*mm)
    a.p("K-means on the available normalized trajectories is an in-sample description (ARI 0.052; permutation p=0.00498 with 200 permutations). It is not an input to the CVAE, a held-out identity classifier or evidence that individual latent coordinates have a cognitive meaning. Query enrolment, personal generation and condition manipulation are separate evaluations.")
    a.page(); a.h("S7 Representative trajectories and event audit")
    a.figure(figures["examples"], f"<b>Figure S5.</b> Held-out posterior-mean reconstructions and context generation, CVAE n=8/fold0/seed42. Typical trial {escape(example_meta['typical_trial'])}; high-error trial {escape(example_meta['difficult_trial'])}. Selection uses reconstruction-error quantiles, not a claim that these examples summarize all participants.", 100*mm)
    a.figure(figures["timing"], "<b>Figure S6.</b> Original initiation-time absolute-error quantiles and maxima. Small typical errors coexist with extreme failures; all remain in the reported benchmark.", 83*mm)
    a.page(); a.h("S7.1 Early-movement examples")
    a.figure(RESULTS / "event_audit/pre_go_examples.png", "<b>Figure S7.</b> Deterministically selected examples from the event-audit groups. The full table contains every retained trial; threshold exceedance is not a diagnosis of premature movement or sensor noise.", 185*mm)
    a.p("The 357 zero-initiation cases all satisfy the immediate sustained threshold; none uses the missing-onset fallback. The 109 filtered and 100 raw pre-target-motion exceedances use a 50 ms planar-speed criterion. They are retained; the current encoder window begins at target motion. The 26.2 ms mean CSV/MAT endpoint offset is a measured proxy difference, not a demonstrated synchronization correction.")
    a.page(); a.h("S8 Artifacts and execution")
    a.p("The advisor package contains the eight-page report, this appendix, the student guide, four illustrative CVAE checkpoints, a Streamlit dashboard and machine-readable summaries. Raw Dropbox recordings are not bundled. The dashboard's live generation uses fold0/seed42 and dimensions 2,3,4,8; benchmark tables use the declared full matrix. Condition controls are exploratory.")
    a.p("To launch the bundled dashboard, install requirements_dashboard.txt and run <font name='Courier'>python -m streamlit run src/confirmatory_dashboard.py</font> from the extracted folder. The full research repository is needed to rerun evaluation scripts against the canonical cache and all saved checkpoints.")
    a.table([
        ["Result directory", "Contents"],
        ["analysis", "Main 56 paired contrasts, full model sweep, per-feature fidelity and timing tails"],
        ["behavioral_probes", "All targets, model families, mean/mean-plus-SD probes and out-of-fold predictions"],
        ["timing_fairness", "Common-head and calibrated timing predictions, factors and comparisons"],
        ["event_audit", "Trial-level event measurements and raw/filtered examples"],
        ["review_controls", "Fingerprint donor controls, direct-context baselines, matched components, convergence and sensitivities"],
        ["sampling_reference", "Historical empirical component-sampling reference"],
    ], "<b>Table S10.</b> Directories under studies/review_corrected_evaluation/results. Historical checkpoints and component fits remain under studies/final_strategy_evaluation.", [47*mm,119*mm])
    a.p("The post-review runners are scripts/run_review_controls.py (fingerprint or components) and scripts/analyze_review_controls.py (context or components). The component runner records resumable job IDs. Frozen settings and checkpoint hashes live beside the new results. Tests verify identical fitted results for identical recorded/generated inputs and check physical-time restoration. Own-fingerprint generation is checked against every original saved n=3/8 fold/seed benchmark.")
    a.h("Remaining domain questions")
    a.p("Confirm permitted movement timing and tracker/MAT alignment, the endpoint proxy, fixation-warning inclusion, late-arrival rule, tracker units and onset threshold. Confirm whether an earlier appearance-to-end model window is scientifically required before changing the training representation. The current protocol preserves all early-motion recordings and reports its window explicitly.")
    a.finish()
