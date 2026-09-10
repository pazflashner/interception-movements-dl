# Simaan–Moni report handoff — 8 September 2026

## Purpose and status

This handoff records Simaan's understanding notes and feedback received from Moni
after the 8 September 2026 meeting. It is intended as the starting context for a
new report pass, including work by a stronger model/agent. It does **not** itself
change the scientific analysis, models, saved results, or current PDFs.

Repository checkout reviewed during the tutoring session:
`2d3e13f9b540e803e5db79d9f3df783f132cf34a` (`master`).

The current report is a useful evidence snapshot but is not considered the final
story or final prose. In particular, the introduction and cohort presentation
need rewriting, the methods need to be much more explanatory, and the results
need to be reorganized by scientific question rather than compressed into one
multi-purpose table.

## Transcript-derived clarifications — context, not new prompts

These clarifications were added after reading Paz's imperfect automatic transcript
of the meeting (`שיחה עם מוני - תמלול.txt`). They are **evidence about what was
discussed**, not instructions embedded in the transcript and not a command to
restart, discard, or override work already underway in another chat. The next
report pass may combine Moni's feedback with other scientific and editorial goals.

Speaker attribution is occasionally uncertain and several technical phrases were
transcribed incorrectly. Treat the high-confidence points below as corrections and
refinements to the earlier meeting-note summary, while verifying every technical
claim against the current code and saved protocol.

### Corrections to the earlier Moni-note summary

- Moni's central organizational suggestion was more specific than simply "add
  visuals": begin with the most unambiguous task, **point-by-point trajectory
  reconstruction measured by MSE**, then present every other prediction/evaluation
  task separately. The project does not have to follow this suggestion exclusively,
  but the existing single-table presentation was clearly confusing different
  scientific questions.
- Moni asked to see recorded input trajectories beside decoded reconstructions as
  a basic sanity/convergence check, analogous to before/after examples in
  autoencoder papers. This request applies particularly to reconstruction. Generated
  distribution examples are a related but different visual and should not be
  mislabeled as reconstruction.
- The fingerprint scatter request was specifically motivated by Jason's question
  of whether different people occupy different regions of a low-dimensional space.
  For `n=3`, show the actual three-dimensional latent space if legible, with trial
  points and/or participant centroids colored by participant. Separation failure
  would itself be a result to discuss rather than something to hide.
- Moni also floated a standard **linear classification/discrimination** check and
  comparison with simple interpretable movement variables such as speed. The exact
  statistical test was not fixed in the meeting. Do not treat "do a linear test"
  as a fully specified analysis or silently choose one without stating the question,
  split, observational unit, and leakage controls.
- Moni suggested exploring higher capacities such as 15 or 20 dimensions to show
  a compression-performance curve, while retaining the scientific relevance of
  Jason's desired two/three-number constraint. This is proposed future work, not a
  current result and not authorization in this handoff to retrain models.
- Moni's overall assessment was positive but conditional: the results looked
  plausible and potentially useful, while the authors' method-level understanding
  and the report's explanation were not yet sufficient. The main problem he
  identified was not that neural methods failed, but that the report did not make
  each task, method, unit, and intermediate representation understandable.

### Technical corrections to approximate statements in the transcript

- Meeting speech referred approximately to a trajectory with 300 numbers and a
  spline representation of "about 20" parameters. The current confirmatory model
  is two-dimensional: `100 x 2 = 200` trajectory values. With five fixed interior
  knots and cubic degree three, the implementation retains exactly nine coefficients
  for x and nine for y, or **18 coefficients per trial**.
- The five spline knots are fixed/equally spaced in the current implementation;
  they are not learned separately for every trial. Each trial's spline coefficients
  are fitted, and the PCA basis is then learned from training-participant coefficient
  vectors only.
- The effective confirmatory Spline+PCA runner uses shape-only coefficients
  (`include_timing=False`). Timing is not part of its PCA input; a separate Timing
  Ridge takes PCA scores plus conditions and predicts initiation/movement duration.
- Meeting shorthand described a fingerprint as averaging all of a person's trials.
  In the frozen evaluation, a held-out participant's fingerprint averages **context
  trial codes only**. Disjoint query trials provide evaluation evidence.
- A transcript phrase appears to say that three dimensions outperform eight, but
  its surrounding discussion says reconstruction MSE decreases substantially at
  eight dimensions. Use the saved tables: both spline and CVAE reconstruct better
  at `n=8` than at `n=3`.
- Enrollment does not prove that participant regions are cleanly separated. It
  encodes query trajectories and classifies their codes against seven context
  centroids. The approximately 40.9% (`n=3`) and 50.7% (`n=8`) balanced accuracies
  exceed 14.3% nominal chance but still show substantial overlap.

### Interpretive cautions from the transcript

- Moni treated Spline+PCA as a strong classical benchmark, not as an unfairly
  privileged input. Both CVAE and Spline+PCA receive a complete trajectory for the
  reconstruction task and compress it to `n` values. Splines contribute a strong
  smooth-curve inductive bias; in generation neither method receives query shapes.
- An absolute MSE cannot be called good or bad from its number alone without scale,
  a comparator, and preferably visual reconstruction examples. The report should
  explain units and show what the numerical error looks like geometrically.
- Moni offered limited-data and weak/non-informative conditioning as possible
  explanations for classical/neural or conditional/unconditional results. These
  were hypotheses, not demonstrated causes. The current study has no learning-curve
  evidence establishing sample count as the cause.
- His AE-versus-VAE interpretation was that a deterministic AE advantage may suggest
  the tested task rewards compression more than stochastic latent regularization.
  Present this as discussion-level interpretation, not a causal result.
- The transcript does not establish that Moni endorsed cognitive meanings for
  latent coordinates, the exact onset/event assumptions, five knots as optimal,
  or the full 11-feature set as coming from Jason's repository.

## Moni's direct feedback from the meeting

### 1. Add clear visual results

- Show actual trajectory outputs against recorded trajectories so that a reader
  can see what reconstruction/generation quality means. Use one or a small number
  of carefully selected, understandable examples.
- Show participants in a scatter plot using their fingerprints, with participant
  colors, so between-participant separation and overlap are visible.
- Preferably accompany the fingerprint visualization with a suitable quantitative
  linear test or other clearly justified analysis of participant differences.
- Do not choose only flattering examples. Define the example-selection rule and
  distinguish illustrative plots from full held-out evidence.

### 2. Do not tell all results through one table

- Columns such as trajectory MSE, timing MAE, and distribution distance answer
  different questions and belong to different analytical paths.
- Present the methods and results as a clean, sequential scientific story, one
  question at a time.
- A complete comparison table can remain in the appendix after the individual
  stories have been explained. It may include p-values and full technical detail,
  but it should not carry the main narrative.
- The main text should explicitly separate reconstruction, timing prediction,
  generation/distribution fidelity, fingerprint controls, behavioural probes,
  and minimum-jerk analysis.

### 3. Put enough method detail in the report to answer likely questions

- The authors must understand every represented method and result.
- Moni specifically asked how Spline+PCA was implemented. The report should answer
  that question directly rather than requiring a reader to interrogate the code.
- Explain concrete inputs, transformations, fitted objects, outputs, evaluation
  units, and data separation for every method.
- Go deeper into method details where needed, while keeping the narrative readable.
  The objective is not maximal jargon; it is to prevent basic methodological
  ambiguity.

### Overall direction

There is a good project and useful evidence here. The next report should make it
cleaner, more visual, more coherent, and more story-driven without overstating
the conclusions.

## Simaan's current conceptual map

The project should be explained as four phases:

1. **Training:** fit representations using training participants, with validation
   participants used for model/hyperparameter selection.
2. **Enrollment/context:** encode approximately half of each held-out test
   participant's trials and average trial codes into a participant fingerprint.
3. **Task-specific evaluation:** separately test reconstruction, timing prediction,
   generation, enrollment, and behavioural-summary prediction.
4. **Scientific comparison:** compare recorded query trials with reconstructions or
   generated samples using metrics appropriate to the particular question.

Methods such as K-means, Spline+PCA, CVAE, controls, behavioural probes, and
minimum-jerk fitting are not necessarily consecutive stages of one pipeline.
Several are alternative models or separate evaluations.

## Scientific question and scope

The tested question is whether a compact code inferred from context trials of a
participant excluded from model training can help generate the distribution of
that participant's remaining, within-session query movements under observed task
conditions.

This is not zero-shot prediction: test participants provide context trials for
enrollment, and generation uses the empirical query-condition mixture. It is not
evaluation on a new external cohort or outside the observed task conditions. Good
generation does not identify a cognitive strategy, prove stable identity, or
demonstrate extrapolation.

Jason Friedman's broader aim includes detailed and variable kinematics across
people: initiation time, movement time, curvature, trajectories, and associated
distributions, with flexibility and eventually reasonable extrapolation. The
current compact generator operationalizes part of that aim; it does not establish
the full aspiration.

## Data representation and assumptions to explain explicitly

- The canonical study contains 4,732 retained condition-2 trials from 28
  participants.
- Marker 5 is target **appearance**, not target motion.
- MAT `dotArray` determines when the target begins moving, 0.20–0.50 s after
  appearance in the audited data.
- MAT `pressedTime` is an experiment-variable name treated by the project as the
  finger-interception/arrival event. The code computes `pressedTime - starttime`.
  It should not be described as the participant pressing a button without domain
  confirmation.
- The model endpoint is the final CSV tracker sample, an arrival proxy averaging
  26.2 ms before MAT `pressedTime` (SD 3.6 ms).
- Positions are regularized on the tracker frame counter, fourth-order zero-phase
  Butterworth filtered at 10 Hz, origin-aligned, and resampled to 100 phase points.
- Full recordings remain 3-D. Finger onset uses post-target-motion **3-D** speed,
  while the final model uses table-plane x-y trajectories.
- Finger onset is the first speed above 5 tracker units/s sustained for three
  frames. Three-frame persistence is intended to avoid a one-frame derivative
  spike, but the exact threshold remains an analysis assumption requiring domain
  validation.
- Initiation duration is target motion to detected finger onset. Movement duration
  is finger onset to the final tracker sample.
- The encoder window is target motion to final sample, so post-motion waiting is
  included. Earlier appearance-to-motion data remain in recordings/audits but do
  not enter the encoder.
- Resampling preserves relative waiting as a fraction of the 100-point shape but
  removes absolute seconds. The two physical durations are not encoder inputs;
  they are decoder outputs supervised through the training loss.
- Raw tracker position units are not confirmed as millimetres or centimetres.

Domain questions still requiring Jason's confirmation include tracker units,
event synchronization, validity of the final-sample endpoint proxy, exact onset
threshold, permitted early movement, fixation-warning inclusion, the late-arrival
rule, and whether an appearance-to-end model window is scientifically required.

## Conditions: exact five-value encoding

The CVAE encoder and decoder receive five task-condition values:

- three one-hot values for `sp` start category;
- one binary side value (`0` left, `1` right);
- one continuous exact executed target speed derived from MAT target motion.

The `sp` category couples target starting position with speed range; these are not
independent categorical axes. Observed dashboard ranges are:

| `sp` | One-hot | Starting position | Executed speed range (screen widths/s) |
|---|---|---:|---:|
| 1 | `[1,0,0]` | 120 mm | 0.4998–0.5880 |
| 2 | `[0,1,0]` | 140 mm | 0.5844–0.6864 |
| 3 | `[0,0,1]` | 160 mm | 0.6666–0.7842 |

Example: category 2, right, executed speed 0.64 is encoded as
`[0, 1, 0, 1, 0.64]`. The screen-width conversion assumes a 1920-pixel width and
still requires confirmation.

## Participant folds, validation, and seeds

Four fixed participant folds each contain 17 training, four validation, and seven
test participants. Within a fold, the participant sets and all their trials are
disjoint. The four seven-person test sets are disjoint, so every participant is
tested exactly once. Across folds, development sets overlap.

- Training participants update fitted parameters.
- Validation participants do not update neural weights through backpropagation;
  validation loss selects checkpoints, controls learning-rate reduction/early
  stopping, and selects Ridge regularization/settings.
- Test participants are reserved for final out-of-fold evaluation.

Each fold therefore requires different fitted models. Seeds 42, 43, and 44 repeat
neural training under different initialization/optimization randomness; they have
no scientific meaning and are averaged rather than selecting the best seed.

## Five model families

1. **CVAE:** primary conditional variational autoencoder/generator.
2. **Spline+PCA:** strong classical smooth-curve and linear-compression benchmark.
3. **Conditional AE:** removes stochastic sampling and KL regularization to test
   the contribution of the variational bottleneck.
4. **Unconditional VAE:** suppresses task-condition inputs to test the contribution
   of explicit conditioning.
5. **Condition Ridge:** predicts shape/timing from conditions only and has no
   participant fingerprint; it tests how far population-level task information
   can go without enrollment.

The controls do not prove a design element is inherently useful or useless. In
particular, adding condition inputs does not guarantee held-out improvement, and
the unconditional VAE remained competitive on several endpoints.

## CVAE: implementation explanation needed in the new report

The CVAE input trajectory is 100 x-y points flattened to 200 coordinate values,
plus the five condition values. Two 256-unit ReLU layers map the input to a
per-trial diagonal Gaussian posterior:

`q(z | trajectory, condition) = Normal(mu, diag(sigma^2))`.

A latent draw samples all `n` coordinates together. The decoder receives that
latent draw **and the explicit conditions** and outputs 200 reconstructed/generated
coordinates plus two log-duration values. It does not directly output curvature,
minimum-jerk components, or behavioural distributions; those are derived later.

The training loss is:

`trajectory squared error + 20 * timing squared error + beta * KL divergence`.

Errors use training-standardized coordinates/log-timings. KL is
Kullback–Leibler divergence between the learned diagonal Gaussian and a standard
Normal prior. `beta` ramps linearly from 0 to 1 over the first 50 epochs to reduce
posterior-collapse risk. The Gaussian posterior/prior and timing weight 20 are
model choices, not discovered biological distributions or uniquely optimal values.

## Spline+PCA: implementation that Moni asked about

This needs a diagram and a worked example in the new report.

For every target-motion-to-end x-y trajectory:

1. Fit separate cubic B-splines to x and y using five interior knots.
2. Retain nine real coefficients per coordinate: `5 knots + degree 3 + 1`, giving
   18 coefficients per trial.
3. Optionally compare raw versus standardized coefficients on validation data;
   the validation reconstruction choice is retained.
4. Fit PCA on **training-participant coefficients only** and retain `n=2,3,4,8`
   PCA scores per trial.
5. Reconstruction applies inverse PCA, restores coefficient scaling, and evaluates
   the spline basis to recover 100 x-y points.

The effective confirmatory runner explicitly uses `include_timing=False`; physical
timings are excluded from PCA. Be careful: the reusable class contains older/default
support and a docstring mentioning timing-inclusive PCA, but the frozen runner and
protocol define the effective shape-only study.

The PCA directions are population-level directions in training coefficient space,
not personal components. A participant-specific spline fingerprint is formed later
by averaging that participant's context-trial PCA scores.

A separate validation-tuned Timing Ridge takes `[PCA scores + five conditions]`
and outputs initiation and movement durations in seconds after inverse log transform.
For generation, codes are sampled around the participant's context-code mean using
training-derived within-person covariance; inverse PCA/splines generate shapes and
Timing Ridge generates durations.

Spline+PCA is a strong, deliberately well-matched classical ML/statistical baseline,
not an unfair source of privileged test information. Both it and CVAE receive a
complete trial during reconstruction and compress it to `n` values. Splines impose
a strong smooth-curve inductive bias. In generation neither model receives query
shapes. Spline's reconstruction advantage must be stated plainly; it does not end
the generative-distribution question.

No documented empirical justification establishes five interior knots as uniquely
optimal. Do not invent a rationale; flag the fixed choice or add a planned sensitivity
analysis if scientifically approved.

## Condition Ridge: exact role and implementation

Condition Ridge is neither a CVAE nor a spline model. Two ordinary Ridge regressions
are fitted using the five condition values per training trial:

- conditions -> 200 flattened trajectory coordinates;
- conditions -> two standardized log-durations.

Validation trials select the Ridge penalty. For generation, the model predicts a
population condition-specific mean and adds a residual sampled from a training trial
with matching categorical start category and side; exact speed stays in the Ridge
prediction. It never receives subject ID, context trials, or a fingerprint.

Its scientific question is whether task conditions plus population residual
variation can match query distributions without personal enrollment. At `n=3`,
CVAE is clearly better for reconstruction and timing but does not significantly
beat Condition Ridge on mean KS, energy, or MMD after the declared correction. At
`n=8`, CVAE beats it on all three main generation distances. This does not justify
saying that conditions are the "main culprit" or that participant variation is
insignificant; the same-decoder control independently shows useful personal
information even at `n=3`.

## Separate evaluation tasks and metrics

### Reconstruction

An actual test trajectory enters the encoder/representation and is reconstructed.
Trajectory MSE is calculated for that same trial across all 200 x-y coordinates,
then averaged over trials within each participant and over participants equally.
Lower MSE means more accurate reconstruction of an observed input; it is not a
generation result.

### Timing prediction

The representation code and conditions produce initiation and movement-duration
predictions. Absolute error is averaged within participants and then across
participants equally. MAE is reported in milliseconds. It is a typical absolute
error, not a maximum bound.

### Generation and distribution fidelity

For a held-out participant, approximately half-session context trials form a
fingerprint. Generation samples 120 latent/PCA codes around it, samples condition
rows from the query condition mixture, and decodes shapes/timings without using
query shapes or timing labels. The same feature function turns every recorded query
trial and generated trial into an 11-value row.

- **KS (Kolmogorov–Smirnov):** computed separately for each feature; the largest
  difference between empirical cumulative distributions. Mean KS averages the 11
  feature-level discrepancies. Zero is ideal.
- **Energy discrepancy:** multivariate comparison of standardized 11-feature
  vectors using between-set and within-set Euclidean distances. Lower is better.
- **MMD squared (maximum mean discrepancy):** multivariate kernel comparison of
  standardized 11-feature vectors. It can detect wrong joint relationships even
  when individual marginals look correct. The unbiased estimate may be slightly
  negative under finite sampling. Lower is better.

Neither Energy nor MMD is accuracy or a causal/coherence label. The recorded query
cloud defines the empirical joint pattern being compared.

The 11 primary features are initiation time, movement time, peak speed, relative
time to peak speed, path length, straight-line distance, curvature index, maximum
lateral deviation, heuristic speed-peak count, and x/y endpoints. The complete
set is not documented as having been selected by Jason's GitHub code. Jason's
repository is specifically the basis for the separate adapted minimum-jerk method.
The primary feature called `n_submovements` internally is a legacy alias for the
heuristic speed-peak count and must not be confused with fitted minimum-jerk count.

### Enrollment

For each seven-person test group, every participant's context codes are averaged
to form seven centroids. Each query trajectory is encoded, scaled by context-code
spread, and assigned to the closest centroid. Balanced accuracy averages recall
across participants. It is calculated over all query trials, not one illustrative
query.

CVAE enrollment is approximately 40.9% at `n=3` and 50.7% at `n=8`, versus nominal
chance `1/7 = 14.3%`. This supports above-chance participant information with
substantial overlap; it is not zero-shot identity recognition or cross-session
stability.

### Statistical comparisons

Main model contrasts use one seed-averaged metric per held-out participant in a
paired two-sided Wilcoxon test. Benjamini–Hochberg correction is applied across
the declared 56-comparison family. Appendix Table S2 prints selected CVAE–spline
contrasts; the complete current table is
`review_evidence/main_paired_model_comparisons.csv`.

Not every endpoint/model comparison is significant. Key CVAE–spline results at
`n=3` and `n=8` are significant in opposite endpoint-specific directions: spline
has lower trajectory reconstruction MSE, while CVAE has lower original-head timing
MAE and lower main generation distances. This is a tradeoff, not a universal winner.

## Current key results to preserve accurately

- Spline reconstructs better than CVAE: at `n=3`, MSE 0.097 versus 0.177; at
  `n=8`, 0.023 versus 0.032.
- CVAE has lower original-head timing MAE and several lower generated-distribution
  distances than spline. Common-head timing tests shrink the initiation advantage;
  movement-time differences do not survive their correction.
- Same-decoder fingerprint controls hold decoder, condition draws, noise, covariance,
  and reference fixed. Own fingerprints beat population and wrong-person controls
  on KS, Energy, and MMD for `n=3/8`; all 12 BH-adjusted p-values are below 0.0004.
- At `n=8`, own/population/wrong mean KS is approximately 0.219/0.290/0.317.
- Direct measured context summaries have lower MAE than latent-code probes on all
  14 tested mean/SD behavioural targets at both `n=3` and `n=8`.
- The `n=3` CVAE mean-initiation probe has pooled R-squared 0.723, but 10/14 pooled
  probe R-squared values are negative. Selective information is the defensible claim.
- K-means is an in-sample exploratory analysis, not a CVAE stage: trajectory
  ARI 0.052 with permutation p=0.00498 indicates weak non-random structure rather
  than clean participant clusters.
- Matched minimum-jerk refitting covers 2,376 recorded query and 1,680 generated
  trajectories. Count total variation is 0.214/0.232 at `n=3/8`, smaller than the
  historical mismatched-procedure result, but all seven discrepancies remain above
  the empirical reference's 95th percentile. This is secondary and assumption-sensitive.
- There is no single best model across reconstruction, timing, and distribution
  endpoints.

## Visuals requested for the next report

### Trajectory examples

Show recorded versus reconstructed trajectories for CVAE and Spline+PCA, preferably
at `n=3` and/or `n=8`. Also show generated samples against the participant's recorded
query distribution, clearly separated from reconstruction. Suggested safeguards:

- predefine selection such as median participant/median-error trial, plus an honest
  difficult example;
- use the same axes, units, origin, and condition labels;
- label whether the plot is reconstruction or generation;
- state that examples are illustrative and pair them with aggregate evidence.

### Fingerprint scatter

- Plot context fingerprints with participant colors.
- `n=2` can be plotted directly. For `n=3/8`, use an explicitly labeled projection
  or small multiples rather than implying the displayed two axes are the whole code.
- Show context centroids and possibly query codes with transparency.
- Pair the picture with a justified quantitative analysis. Candidate analyses must
  respect folds and repeated trials; avoid a naive row-level test that treats all
  within-person trials as independent.
- Existing enrollment and K-means results can inform the plot but answer different
  questions. Do not present ARI, enrollment, or a proposed linear test as interchangeable.

The exact "linear test" requested by Moni is not yet operationally specified. Decide
whether the intended question is linear separability/classification, between-person
variance, or a participant effect in latent space before choosing a test. This is a
scientific/statistical design decision, not just a plotting task.

## Recommended new-report narrative

1. Scientific motivation and the exact tested question.
2. Data/events/model window, with a simple timing diagram.
3. Evaluation protocol: participant folds and context/query enrollment.
4. Primary CVAE, explained with an input-to-output diagram and loss.
5. Strong classical Spline+PCA comparator, with a worked trajectory example.
6. Question 1 — reconstruction: qualitative plots, MSE, paired inference.
7. Question 2 — physical timing: original heads, common-head sensitivity, MAE.
8. Question 3 — generation: recorded/generated examples, feature extraction,
   KS/Energy/MMD.
9. Question 4 — does personal context matter? Same-decoder controls.
10. Question 5 — what is captured by the fingerprint? Enrollment, probes,
    direct-context negative control, and fingerprint visualization.
11. Secondary minimum-jerk analysis, clearly separated from primary features.
12. Limitations, domain questions, and scoped conclusion.
13. Appendix: full five-family matrix, complete p-values, implementation details,
    sensitivity results, and reproducibility artifacts.

K-means can appear early as exploratory motivation or later as a diagnostic, but
must be labeled in-sample and weak. It should not be drawn as a preprocessing stage
or used to imply the CVAE discovered clean subject clusters.

## Questions for the next report pass / advisors

### For Moni or the report-design decision

- What exact inferential question should accompany the fingerprint scatter:
  linear separability, participant effect, or another measure?
- Which visual examples best communicate performance without cherry-picking?
- Should the primary story lead with the desired `n=2/3` compact interface or use
  `n=8` as the clearest evidence of higher-capacity generation?
- How should the competitive unconditional VAE and `n=3` Condition Ridge result be
  framed in the discussion?
- Is direct feature/distribution prediction a useful future comparator if detailed
  trajectory noise is not scientifically essential?
- Are the 11 primary features an adequate operationalization of the scientific aim,
  and what rationale/source should be supplied for their selection?
- Should five spline knots receive a sensitivity analysis or simply be disclosed as
  a fixed benchmark choice?

### For Jason/domain confirmation

- Confirm tracker coordinate units and the 1920-pixel target-speed normalization.
- Confirm the semantics/alignment of `pressedTime`, target motion, and tracker end.
- Confirm the endpoint proxy, onset threshold, pre-target-motion behavior, and
  whether the current target-motion-to-end window is scientifically appropriate.
- Confirm treatment of fixation warnings, early outcomes, and the late-arrival rule.
- Confirm that the adapted minimum-jerk constraints and selected outputs match the
  intended scientific use.

## High-value source anchors for the next AI

- Current paper authoring: `reports/concise_article.py`
- Preprocessing/events: `src/preprocessing.py`, `src/data_loading.py`,
  `src/trajectory_view.py`
- CVAE/conditions/timing/loss: `src/vae_model.py`
- Effective CVAE runner: `scripts/run_confirmatory_cvae.py`
- Spline representation: `src/baseline_spline.py`
- Effective shape-only spline runner: `scripts/run_confirmatory_spline.py`
- Spline Timing Ridge/generation: `src/confirmatory_spline.py`
- Condition Ridge: `src/confirmatory_controls.py`
- Folds: `src/confirmatory_protocol.py`
- Context/query, enrollment, distances: `src/context_query.py`
- Feature formulas: `src/features.py`
- Current corrected protocol: `studies/review_corrected_evaluation/PROTOCOL.md`
- Compact saved evidence: `review_evidence/`
- Existing report, appendix, and guide: `output/pdf/`

Handoff files and earlier AI conclusions are navigation aids. Every new statement,
visual, and result should be verified against the effective runner, frozen protocol,
saved participant-level results, and current source functions before publication.
